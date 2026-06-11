import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { SectionHeader } from "@/components/section-header"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Mail01Icon,
  NotificationIcon,
  Settings01Icon,
  UserIcon,
} from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"

export function SettingsPage() {
  return (
    <div className="space-y-4">
      <SectionHeader
        eyebrow="Account"
        title="Settings"
        description="Profile, notifications, and display preferences."
      />

      <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>
                <HugeiconsIcon
                  icon={UserIcon}
                  strokeWidth={1.5}
                  className="mr-1 inline size-4"
                />
                Profile
              </CardTitle>
              <CardDescription>
                How others see you in the operator console.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                <Avatar className="size-12">
                  <AvatarFallback className="bg-primary text-primary-foreground">
                    DB
                  </AvatarFallback>
                </Avatar>
                <div>
                  <p className="font-medium">Diar B.</p>
                  <p className="text-xs text-muted-foreground">
                    Operator · Astana
                  </p>
                </div>
                <Button variant="outline" size="sm" className="ml-auto">
                  Change photo
                </Button>
              </div>
              <Separator className="my-4" />
              <FieldGroup className="grid gap-4 md:grid-cols-2">
                <Field>
                  <FieldLabel htmlFor="first-name">First name</FieldLabel>
                  <Input id="first-name" defaultValue="Diar" />
                </Field>
                <Field>
                  <FieldLabel htmlFor="last-name">Last name</FieldLabel>
                  <Input id="last-name" defaultValue="Begisbayev" />
                </Field>
                <Field className="md:col-span-2">
                  <FieldLabel htmlFor="email">
                    <HugeiconsIcon
                      icon={Mail01Icon}
                      strokeWidth={1.5}
                      className="mr-1 inline size-3.5"
                    />
                    Email
                  </FieldLabel>
                  <Input
                    id="email"
                    type="email"
                    defaultValue="diar@michi.local"
                  />
                </Field>
              </FieldGroup>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>
                <HugeiconsIcon
                  icon={NotificationIcon}
                  strokeWidth={1.5}
                  className="mr-1 inline size-4"
                />
                Notifications
              </CardTitle>
              <CardDescription>
                When and how you want to be notified.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                {
                  id: "high",
                  title: "High-severity alerts",
                  desc: "Push + email.",
                  on: true,
                },
                {
                  id: "med",
                  title: "Medium-severity alerts",
                  desc: "In-app only.",
                  on: true,
                },
                {
                  id: "weekly",
                  title: "Weekly executive summary",
                  desc: "Email, Mondays at 09:00.",
                  on: false,
                },
                {
                  id: "model",
                  title: "Model drift warnings",
                  desc: "Notify on PSI > 0.2.",
                  on: true,
                },
              ].map((n) => (
                <div
                  key={n.id}
                  className="flex items-center justify-between rounded-2xl bg-muted/40 p-3"
                >
                  <div>
                    <Label htmlFor={n.id}>{n.title}</Label>
                    <p className="text-xs text-muted-foreground">{n.desc}</p>
                  </div>
                  <Switch id={n.id} defaultChecked={n.on} />
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>
              <HugeiconsIcon
                icon={Settings01Icon}
                strokeWidth={1.5}
                className="mr-1 inline size-4"
              />
              Display
            </CardTitle>
            <CardDescription>Theme, density, and time format.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-2xl bg-muted/40 p-3">
              <Label>Theme</Label>
              <p className="mb-2 text-xs text-muted-foreground">
                Use the theme toggle in the sidebar to switch.
              </p>
            </div>
            <div className="rounded-2xl bg-muted/40 p-3">
              <Label htmlFor="tz">Time zone</Label>
              <Input id="tz" defaultValue="Asia/Almaty (UTC+6)" />
            </div>
            <div className="flex items-center justify-between rounded-2xl bg-muted/40 p-3">
              <div>
                <Label htmlFor="compact">Compact density</Label>
                <p className="text-xs text-muted-foreground">
                  Reduce vertical padding.
                </p>
              </div>
              <Switch id="compact" />
            </div>
            <Button className="w-full">Save changes</Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
